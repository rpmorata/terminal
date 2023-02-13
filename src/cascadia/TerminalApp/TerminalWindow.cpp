// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "pch.h"
#include "TerminalWindow.h"
#include "../inc/WindowingBehavior.h"
#include "TerminalWindow.g.cpp"
#include "SettingsLoadEventArgs.g.cpp"

#include <LibraryResources.h>
#include <WtExeUtils.h>
#include <wil/token_helpers.h>

#include "../../types/inc/utils.hpp"

using namespace winrt::Windows::ApplicationModel;
using namespace winrt::Windows::ApplicationModel::DataTransfer;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;
using namespace winrt::Windows::UI::Core;
using namespace winrt::Windows::System;
using namespace winrt::Microsoft::Terminal;
using namespace winrt::Microsoft::Terminal::Control;
using namespace winrt::Microsoft::Terminal::Settings::Model;
using namespace ::TerminalApp;

namespace winrt
{
    namespace MUX = Microsoft::UI::Xaml;
    namespace WUX = Windows::UI::Xaml;
    using IInspectable = Windows::Foundation::IInspectable;
}

// !!! IMPORTANT !!!
// Make sure that these keys are in the same order as the
// SettingsLoadWarnings/Errors enum is!
static const std::array settingsLoadWarningsLabels{
    USES_RESOURCE(L"MissingDefaultProfileText"),
    USES_RESOURCE(L"DuplicateProfileText"),
    USES_RESOURCE(L"UnknownColorSchemeText"),
    USES_RESOURCE(L"InvalidBackgroundImage"),
    USES_RESOURCE(L"InvalidIcon"),
    USES_RESOURCE(L"AtLeastOneKeybindingWarning"),
    USES_RESOURCE(L"TooManyKeysForChord"),
    USES_RESOURCE(L"MissingRequiredParameter"),
    USES_RESOURCE(L"FailedToParseCommandJson"),
    USES_RESOURCE(L"FailedToWriteToSettings"),
    USES_RESOURCE(L"InvalidColorSchemeInCmd"),
    USES_RESOURCE(L"InvalidSplitSize"),
    USES_RESOURCE(L"FailedToParseStartupActions"),
    USES_RESOURCE(L"FailedToParseSubCommands"),
    USES_RESOURCE(L"UnknownTheme"),
    USES_RESOURCE(L"DuplicateRemainingProfilesEntry"),
};
static_assert(settingsLoadWarningsLabels.size() == static_cast<size_t>(SettingsLoadWarnings::WARNINGS_SIZE));
// Errors are defined in AppLogic.cpp

// Function Description:
// - General-purpose helper for looking up a localized string for a
//   warning/error. First will look for the given key in the provided map of
//   keys->strings, where the values in the map are ResourceKeys. If it finds
//   one, it will lookup the localized string from that ResourceKey.
// - If it does not find a key, it'll return an empty string
// Arguments:
// - key: the value to use to look for a resource key in the given map
// - map: A map of keys->Resource keys.
// Return Value:
// - the localized string for the given type, if it exists.
template<typename T>
winrt::hstring _GetMessageText(uint32_t index, const T& keys)
{
    if (index < keys.size())
    {
        return GetLibraryResourceString(til::at(keys, index));
    }
    return {};
}

// Function Description:
// - Gets the text from our ResourceDictionary for the given
//   SettingsLoadWarning. If there is no such text, we'll return nullptr.
// - The warning should have an entry in settingsLoadWarningsLabels.
// Arguments:
// - warning: the SettingsLoadWarnings value to get the localized text for.
// Return Value:
// - localized text for the given warning
static winrt::hstring _GetWarningText(SettingsLoadWarnings warning)
{
    return _GetMessageText(static_cast<uint32_t>(warning), settingsLoadWarningsLabels);
}

// Function Description:
// - Creates a Run of text to display an error message. The text is yellow or
//   red for dark/light theme, respectively.
// Arguments:
// - text: The text of the error message.
// - resources: The application's resource loader.
// Return Value:
// - The fully styled text run.
static Documents::Run _BuildErrorRun(const winrt::hstring& text, const ResourceDictionary& resources)
{
    Documents::Run textRun;
    textRun.Text(text);

    // Color the text red (light theme) or yellow (dark theme) based on the system theme
    auto key = winrt::box_value(L"ErrorTextBrush");
    if (resources.HasKey(key))
    {
        auto g = resources.Lookup(key);
        auto brush = g.try_as<winrt::Windows::UI::Xaml::Media::Brush>();
        textRun.Foreground(brush);
    }

    return textRun;
}

namespace winrt::TerminalApp::implementation
{
    TerminalWindow::TerminalWindow(const TerminalApp::SettingsLoadEventArgs& settingsLoadedResult) :
        _settings{ settingsLoadedResult.NewSettings() },
        _initialLoadResult{ settingsLoadedResult }
    {
        // For your own sanity, it's better to do setup outside the ctor.
        // If you do any setup in the ctor that ends up throwing an exception,
        // then it might look like App just failed to activate, which will
        // cause you to chase down the rabbit hole of "why is App not
        // registered?" when it definitely is.

        // The TerminalPage has to be constructed during our construction, to
        // make sure that there's a terminal page for callers of
        // SetTitleBarContent
        _isElevated = ::Microsoft::Console::Utils::IsElevated();
    }

    // Method Description:
    // - Implements the IInitializeWithWindow interface from shobjidl_core.
    HRESULT TerminalWindow::Initialize(HWND hwnd)
    {
        _root = winrt::make_self<TerminalPage>();
        _root->WindowProperties(*this);
        _dialog = ContentDialog{};

        // Pass commandline args into the TerminalPage. If we were supposed to
        // load from a persisted layout, do that instead.
        auto foundLayout = false;
        if (const auto& layout = LoadPersistedLayout())
        {
            if (layout.TabLayout().Size() > 0)
            {
                std::vector<Settings::Model::ActionAndArgs> actions;
                for (const auto& a : layout.TabLayout())
                {
                    actions.emplace_back(a);
                }
                _root->SetStartupActions(actions);
                foundLayout = true;
            }
        }
        if (!foundLayout)
        {
            _root->SetStartupActions(_appArgs.GetStartupActions());
        }

        // Check if we were started as a COM server for inbound connections of console sessions
        // coming out of the operating system default application feature. If so,
        // tell TerminalPage to start the listener as we have to make sure it has the chance
        // to register a handler to hear about the requests first and is all ready to receive
        // them before the COM server registers itself. Otherwise, the request might come
        // in and be routed to an event with no handlers or a non-ready Page.
        if (_appArgs.IsHandoffListener())
        {
            _root->SetInboundListener(true);
        }

        return _root->Initialize(hwnd);
    }
    // Method Description:
    // - Called around the codebase to discover if this is a UWP where we need to turn off specific settings.
    // Arguments:
    // - <none> - reports internal state
    // Return Value:
    // - True if UWP, false otherwise.
    bool TerminalWindow::IsUwp() const noexcept
    {
        return _isUwp;
    }

    // Method Description:
    // - Called around the codebase to discover if Terminal is running elevated
    // Arguments:
    // - <none> - reports internal state
    // Return Value:
    // - True if elevated, false otherwise.
    bool TerminalWindow::IsElevated() const noexcept
    {
        return _isElevated;
    }

    // Method Description:
    // - Called by UWP context invoker to let us know that we may have to change some of our behaviors
    //   for being a UWP
    // Arguments:
    // - <none> (sets to UWP = true, one way change)
    // Return Value:
    // - <none>
    void TerminalWindow::RunAsUwp()
    {
        _isUwp = true;
    }

    // Method Description:
    // - Build the UI for the terminal app. Before this method is called, it
    //   should not be assumed that the TerminalApp is usable. The Settings
    //   should be loaded before this is called, either with LoadSettings or
    //   GetLaunchDimensions (which will call LoadSettings)
    // Arguments:
    // - <none>
    // Return Value:
    // - <none>
    void TerminalWindow::Create()
    {
        _root->DialogPresenter(*this);

        // In UWP mode, we cannot handle taking over the title bar for tabs,
        // so this setting is overridden to false no matter what the preference is.
        if (_isUwp)
        {
            _settings.GlobalSettings().ShowTabsInTitlebar(false);
        }

        // Pay attention, that even if some command line arguments were parsed (like launch mode),
        // we will not use the startup actions from settings.
        // While this simplifies the logic, we might want to reconsider this behavior in the future.
        if (!_hasCommandLineArguments && _gotSettingsStartupActions)
        {
            _root->SetStartupActions(_settingsStartupArgs);
        }

        _root->SetSettings(_settings, false);
        _root->Loaded({ this, &TerminalWindow::_OnLoaded });
        _root->Initialized([this](auto&&, auto&&) {
            // GH#288 - When we finish initialization, if the user wanted us
            // launched _fullscreen_, toggle fullscreen mode. This will make sure
            // that the window size is _first_ set up as something sensible, so
            // leaving fullscreen returns to a reasonable size.
            const auto launchMode = this->GetLaunchMode();
            if (IsQuakeWindow() || WI_IsFlagSet(launchMode, LaunchMode::FocusMode))
            {
                _root->SetFocusMode(true);
            }

            // The IslandWindow handles (creating) the maximized state
            // we just want to record it here on the page as well.
            if (WI_IsFlagSet(launchMode, LaunchMode::MaximizedMode))
            {
                _root->Maximized(true);
            }

            if (WI_IsFlagSet(launchMode, LaunchMode::FullscreenMode) && !IsQuakeWindow())
            {
                _root->SetFullscreen(true);
            }
        });
        _root->Create();

        _RefreshThemeRoutine();

        auto args = winrt::make_self<SystemMenuChangeArgs>(RS_(L"SettingsMenuItem"),
                                                           SystemMenuChangeAction::Add,
                                                           SystemMenuItemHandler(this, &TerminalWindow::_OpenSettingsUI));
        _SystemMenuChangeRequestedHandlers(*this, *args);

        TraceLoggingWrite(
            g_hTerminalAppProvider,
            "WindowCreated",
            TraceLoggingDescription("Event emitted when the window is started"),
            TraceLoggingKeyword(MICROSOFT_KEYWORD_MEASURES),
            TelemetryPrivacyDataTag(PDT_ProductAndServiceUsage));
    }
    void TerminalWindow::Quit()
    {
        if (_root)
        {
            _root->CloseWindow(true);
        }
    }

    winrt::Windows::UI::Xaml::ElementTheme TerminalWindow::GetRequestedTheme()
    {
        return Theme().RequestedTheme();
    }

    bool TerminalWindow::GetShowTabsInTitlebar()
    {
        return _settings.GlobalSettings().ShowTabsInTitlebar();
    }

    bool TerminalWindow::GetInitialAlwaysOnTop()
    {
        return _settings.GlobalSettings().AlwaysOnTop();
    }

    bool TerminalWindow::GetMinimizeToNotificationArea()
    {
        return _settings.GlobalSettings().MinimizeToNotificationArea();
    }

    bool TerminalWindow::GetAlwaysShowNotificationIcon()
    {
        return _settings.GlobalSettings().AlwaysShowNotificationIcon();
    }
    bool TerminalWindow::RequestsTrayIcon()
    {
        return _settings.GlobalSettings().AlwaysShowNotificationIcon() ||
               _settings.GlobalSettings().MinimizeToNotificationArea() ||
               IsQuakeWindow();
    }

    bool TerminalWindow::GetShowTitleInTitlebar()
    {
        return _settings.GlobalSettings().ShowTitleInTitlebar();
    }

    Microsoft::Terminal::Settings::Model::Theme TerminalWindow::Theme()
    {
        return _settings.GlobalSettings().CurrentTheme();
    }
    // Method Description:
    // - Show a ContentDialog with buttons to take further action. Uses the
    //   FrameworkElements provided as the title and content of this dialog, and
    //   displays buttons (or a single button). Two buttons (primary and secondary)
    //   will be displayed if this is an warning dialog for closing the terminal,
    //   this allows the users to abandon the closing action. Otherwise, a single
    //   close button will be displayed.
    // - Only one dialog can be visible at a time. If another dialog is visible
    //   when this is called, nothing happens.
    // Arguments:
    // - dialog: the dialog object that is going to show up
    // Return value:
    // - an IAsyncOperation with the dialog result
    winrt::Windows::Foundation::IAsyncOperation<ContentDialogResult> TerminalWindow::ShowDialog(winrt::WUX::Controls::ContentDialog dialog)
    {
        // DON'T release this lock in a wil::scope_exit. The scope_exit will get
        // called when we await, which is not what we want.
        std::unique_lock lock{ _dialogLock, std::try_to_lock };
        if (!lock)
        {
            // Another dialog is visible.
            co_return ContentDialogResult::None;
        }

        _dialog = dialog;

        // IMPORTANT: This is necessary as documented in the ContentDialog MSDN docs.
        // Since we're hosting the dialog in a Xaml island, we need to connect it to the
        // xaml tree somehow.
        dialog.XamlRoot(_root->XamlRoot());

        // IMPORTANT: Set the requested theme of the dialog, because the
        // PopupRoot isn't directly in the Xaml tree of our root. So the dialog
        // won't inherit our RequestedTheme automagically.
        // GH#5195, GH#3654 Because we cannot set RequestedTheme at the application level,
        // we occasionally run into issues where parts of our UI end up themed incorrectly.
        // Dialogs, for example, live under a different Xaml root element than the rest of
        // our application. This makes our popup menus and buttons "disappear" when the
        // user wants Terminal to be in a different theme than the rest of the system.
        // This hack---and it _is_ a hack--walks up a dialog's ancestry and forces the
        // theme on each element up to the root. We're relying a bit on Xaml's implementation
        // details here, but it does have the desired effect.
        // It's not enough to set the theme on the dialog alone.
        auto themingLambda{ [this](const Windows::Foundation::IInspectable& sender, const RoutedEventArgs&) {
            auto theme{ _settings.GlobalSettings().CurrentTheme() };
            auto requestedTheme{ theme.RequestedTheme() };
            auto element{ sender.try_as<winrt::Windows::UI::Xaml::FrameworkElement>() };
            while (element)
            {
                element.RequestedTheme(requestedTheme);
                element = element.Parent().try_as<winrt::Windows::UI::Xaml::FrameworkElement>();
            }
        } };

        themingLambda(dialog, nullptr); // if it's already in the tree
        auto loadedRevoker{ dialog.Loaded(winrt::auto_revoke, themingLambda) }; // if it's not yet in the tree

        // Display the dialog.
        co_return co_await dialog.ShowAsync(Controls::ContentDialogPlacement::Popup);

        // After the dialog is dismissed, the dialog lock (held by `lock`) will
        // be released so another can be shown
    }

    // Method Description:
    // - Dismiss the (only) visible ContentDialog
    void TerminalWindow::DismissDialog()
    {
        if (auto localDialog = std::exchange(_dialog, nullptr))
        {
            localDialog.Hide();
        }
    }

    // Method Description:
    // - Displays a dialog for errors found while loading or validating the
    //   settings. Uses the resources under the provided  title and content keys
    //   as the title and first content of the dialog, then also displays a
    //   message for whatever exception was found while validating the settings.
    // - Only one dialog can be visible at a time. If another dialog is visible
    //   when this is called, nothing happens. See ShowDialog for details
    // Arguments:
    // - titleKey: The key to use to lookup the title text from our resources.
    // - contentKey: The key to use to lookup the content text from our resources.
    void TerminalWindow::_ShowLoadErrorsDialog(const winrt::hstring& titleKey,
                                               const winrt::hstring& contentKey,
                                               HRESULT settingsLoadedResult,
                                               const winrt::hstring& exceptionText)
    {
        auto title = GetLibraryResourceString(titleKey);
        auto buttonText = RS_(L"Ok");

        Controls::TextBlock warningsTextBlock;
        // Make sure you can copy-paste
        warningsTextBlock.IsTextSelectionEnabled(true);
        // Make sure the lines of text wrap
        warningsTextBlock.TextWrapping(TextWrapping::Wrap);

        winrt::Windows::UI::Xaml::Documents::Run errorRun;
        const auto errorLabel = GetLibraryResourceString(contentKey);
        errorRun.Text(errorLabel);
        warningsTextBlock.Inlines().Append(errorRun);
        warningsTextBlock.Inlines().Append(Documents::LineBreak{});

        if (FAILED(settingsLoadedResult))
        {
            if (!exceptionText.empty())
            {
                warningsTextBlock.Inlines().Append(_BuildErrorRun(exceptionText,
                                                                  winrt::WUX::Application::Current().as<::winrt::TerminalApp::App>().Resources()));
                warningsTextBlock.Inlines().Append(Documents::LineBreak{});
            }
        }

        // Add a note that we're using the default settings in this case.
        winrt::Windows::UI::Xaml::Documents::Run usingDefaultsRun;
        const auto usingDefaultsText = RS_(L"UsingDefaultSettingsText");
        usingDefaultsRun.Text(usingDefaultsText);
        warningsTextBlock.Inlines().Append(Documents::LineBreak{});
        warningsTextBlock.Inlines().Append(usingDefaultsRun);

        Controls::ContentDialog dialog;
        dialog.Title(winrt::box_value(title));
        dialog.Content(winrt::box_value(warningsTextBlock));
        dialog.CloseButtonText(buttonText);
        dialog.DefaultButton(Controls::ContentDialogButton::Close);

        ShowDialog(dialog);
    }

    // Method Description:
    // - Displays a dialog for warnings found while loading or validating the
    //   settings. Displays messages for whatever warnings were found while
    //   validating the settings.
    // - Only one dialog can be visible at a time. If another dialog is visible
    //   when this is called, nothing happens. See ShowDialog for details
    void TerminalWindow::_ShowLoadWarningsDialog(const Windows::Foundation::Collections::IVector<SettingsLoadWarnings>& warnings)
    {
        auto title = RS_(L"SettingsValidateErrorTitle");
        auto buttonText = RS_(L"Ok");

        Controls::TextBlock warningsTextBlock;
        // Make sure you can copy-paste
        warningsTextBlock.IsTextSelectionEnabled(true);
        // Make sure the lines of text wrap
        warningsTextBlock.TextWrapping(TextWrapping::Wrap);

        for (const auto& warning : warnings)
        {
            // Try looking up the warning message key for each warning.
            const auto warningText = _GetWarningText(warning);
            if (!warningText.empty())
            {
                warningsTextBlock.Inlines().Append(_BuildErrorRun(warningText, winrt::WUX::Application::Current().as<::winrt::TerminalApp::App>().Resources()));
                warningsTextBlock.Inlines().Append(Documents::LineBreak{});
            }
        }

        Controls::ContentDialog dialog;
        dialog.Title(winrt::box_value(title));
        dialog.Content(winrt::box_value(warningsTextBlock));
        dialog.CloseButtonText(buttonText);
        dialog.DefaultButton(Controls::ContentDialogButton::Close);

        ShowDialog(dialog);
    }

    // Method Description:
    // - Triggered when the application is finished loading. If we failed to load
    //   the settings, then this will display the error dialog. This is done
    //   here instead of when loading the settings, because we need our UI to be
    //   visible to display the dialog, and when we're loading the settings,
    //   the UI might not be visible yet.
    // Arguments:
    // - <unused>
    void TerminalWindow::_OnLoaded(const IInspectable& /*sender*/,
                                   const RoutedEventArgs& /*eventArgs*/)
    {
        if (_settings.GlobalSettings().InputServiceWarning())
        {
            const auto keyboardServiceIsDisabled = !_IsKeyboardServiceEnabled();
            if (keyboardServiceIsDisabled)
            {
                _root->ShowKeyboardServiceWarning();
            }
        }

        const auto& settingsLoadedResult = gsl::narrow_cast<HRESULT>(_initialLoadResult.Result());
        if (FAILED(settingsLoadedResult))
        {
            const winrt::hstring titleKey = USES_RESOURCE(L"InitialJsonParseErrorTitle");
            const winrt::hstring textKey = USES_RESOURCE(L"InitialJsonParseErrorText");
            _ShowLoadErrorsDialog(titleKey, textKey, settingsLoadedResult, _initialLoadResult.ExceptionText());
        }
        else if (settingsLoadedResult == S_FALSE)
        {
            _ShowLoadWarningsDialog(_initialLoadResult.Warnings());
        }
    }

    // Method Description:
    // - Helper for determining if the "Touch Keyboard and Handwriting Panel
    //   Service" is enabled. If it isn't, we want to be able to display a
    //   warning to the user, because they won't be able to type in the
    //   Terminal.
    // Return Value:
    // - true if the service is enabled, or if we fail to query the service. We
    //   return true in that case, to be less noisy (though, that is unexpected)
    bool TerminalWindow::_IsKeyboardServiceEnabled()
    {
        if (IsUwp())
        {
            return true;
        }

        // If at any point we fail to open the service manager, the service,
        // etc, then just quick return true to disable the dialog. We'd rather
        // not be noisy with this dialog if we failed for some reason.

        // Open the service manager. This will return 0 if it failed.
        wil::unique_schandle hManager{ OpenSCManagerW(nullptr, nullptr, 0) };

        if (LOG_LAST_ERROR_IF(!hManager.is_valid()))
        {
            return true;
        }

        // Get a handle to the keyboard service
        wil::unique_schandle hService{ OpenServiceW(hManager.get(), TabletInputServiceKey.data(), SERVICE_QUERY_STATUS) };

        // Windows 11 doesn't have a TabletInputService.
        // (It was renamed to TextInputManagementService, because people kept thinking that a
        // service called "tablet-something" is system-irrelevant on PCs and can be disabled.)
        if (!hService.is_valid())
        {
            return true;
        }

        // Get the current state of the service
        SERVICE_STATUS status{ 0 };
        if (!LOG_IF_WIN32_BOOL_FALSE(QueryServiceStatus(hService.get(), &status)))
        {
            return true;
        }

        const auto state = status.dwCurrentState;
        return (state == SERVICE_RUNNING || state == SERVICE_START_PENDING);
    }

    // Method Description:
    // - Get the size in pixels of the client area we'll need to launch this
    //   terminal app. This method will use the default profile's settings to do
    //   this calculation, as well as the _system_ dpi scaling. See also
    //   TermControl::GetProposedDimensions.
    // Arguments:
    // - <none>
    // Return Value:
    // - a point containing the requested dimensions in pixels.
    winrt::Windows::Foundation::Size TerminalWindow::GetLaunchDimensions(uint32_t dpi)
    {
        winrt::Windows::Foundation::Size proposedSize{};

        const auto scale = static_cast<float>(dpi) / static_cast<float>(USER_DEFAULT_SCREEN_DPI);
        if (const auto layout = LoadPersistedLayout())
        {
            if (layout.InitialSize())
            {
                proposedSize = layout.InitialSize().Value();
                // The size is saved as a non-scaled real pixel size,
                // so we need to scale it appropriately.
                proposedSize.Height = proposedSize.Height * scale;
                proposedSize.Width = proposedSize.Width * scale;
            }
        }

        if (_appArgs.GetSize().has_value() || (proposedSize.Width == 0 && proposedSize.Height == 0))
        {
            // Use the default profile to determine how big of a window we need.
            const auto settings{ TerminalSettings::CreateWithNewTerminalArgs(_settings, nullptr, nullptr) };

            const til::size emptySize{};
            const auto commandlineSize = _appArgs.GetSize().value_or(emptySize);
            proposedSize = TermControl::GetProposedDimensions(settings.DefaultSettings(),
                                                              dpi,
                                                              commandlineSize.width,
                                                              commandlineSize.height);
        }

        // GH#2061 - If the global setting "Always show tab bar" is
        // set or if "Show tabs in title bar" is set, then we'll need to add
        // the height of the tab bar here.
        if (_settings.GlobalSettings().ShowTabsInTitlebar())
        {
            // In the past, we used to actually instantiate a TitlebarControl
            // and use Measure() to determine the DesiredSize of the control, to
            // reserve exactly what we'd need.
            //
            // We can't do that anymore, because this is now called _before_
            // we've initialized XAML for this thread. We can't start XAML till
            // we have an HWND, and we can't finish creating the window till we
            // know how big it should be.
            //
            // Instead, we'll just hardcode how big the titlebar chould be. If
            // the titlebar / tab row ever change size, these numbers will have
            // to change accordingly.

            static constexpr auto titlebarHeight = 40;
            proposedSize.Height += (titlebarHeight)*scale;
        }
        else if (_settings.GlobalSettings().AlwaysShowTabs())
        {
            // Same comment as above, but with a TabRowControl.
            //
            // A note from before: For whatever reason, there's about 10px of
            // unaccounted-for space in the application. I couldn't tell you
            // where these 10px are coming from, but they need to be included in
            // this math.
            static constexpr auto tabRowHeight = 32;
            proposedSize.Height += (tabRowHeight + 10) * scale;
        }

        return proposedSize;
    }

    // Method Description:
    // - Get the launch mode in json settings file. Now there
    //   two launch mode: default, maximized. Default means the window
    //   will launch according to the launch dimensions provided. Maximized
    //   means the window will launch as a maximized window
    // Arguments:
    // - <none>
    // Return Value:
    // - LaunchMode enum that indicates the launch mode
    LaunchMode TerminalWindow::GetLaunchMode()
    {
        // GH#4620/#5801 - If the user passed --maximized or --fullscreen on the
        // commandline, then use that to override the value from the settings.
        const auto valueFromSettings = _settings.GlobalSettings().LaunchMode();
        const auto valueFromCommandlineArgs = _appArgs.GetLaunchMode();
        if (const auto layout = LoadPersistedLayout())
        {
            if (layout.LaunchMode())
            {
                return layout.LaunchMode().Value();
            }
        }
        return valueFromCommandlineArgs.has_value() ?
                   valueFromCommandlineArgs.value() :
                   valueFromSettings;
    }

    // Method Description:
    // - Get the user defined initial position from Json settings file.
    //   This position represents the top left corner of the Terminal window.
    //   This setting is optional, if not provided, we will use the system
    //   default size, which is provided in IslandWindow::MakeWindow.
    // Arguments:
    // - defaultInitialX: the system default x coordinate value
    // - defaultInitialY: the system default y coordinate value
    // Return Value:
    // - a point containing the requested initial position in pixels.
    TerminalApp::InitialPosition TerminalWindow::GetInitialPosition(int64_t defaultInitialX, int64_t defaultInitialY)
    {
        auto initialPosition{ _settings.GlobalSettings().InitialPosition() };

        if (const auto layout = LoadPersistedLayout())
        {
            if (layout.InitialPosition())
            {
                initialPosition = layout.InitialPosition().Value();
            }
        }

        // Commandline args trump everything else
        if (_appArgs.GetPosition().has_value())
        {
            initialPosition = _appArgs.GetPosition().value();
        }

        return {
            initialPosition.X ? initialPosition.X.Value() : defaultInitialX,
            initialPosition.Y ? initialPosition.Y.Value() : defaultInitialY
        };
    }

    bool TerminalWindow::CenterOnLaunch()
    {
        // If the position has been specified on the commandline, don't center on launch
        return _settings.GlobalSettings().CenterOnLaunch() && !_appArgs.GetPosition().has_value();
    }

    // Method Description:
    // - See Pane::CalcSnappedDimension
    float TerminalWindow::CalcSnappedDimension(const bool widthOrHeight, const float dimension) const
    {
        return _root->CalcSnappedDimension(widthOrHeight, dimension);
    }

    // Method Description:
    // - Update the current theme of the application. This will trigger our
    //   RequestedThemeChanged event, to have our host change the theme of the
    //   root of the application.
    // Arguments:
    // - newTheme: The ElementTheme to apply to our elements.
    void TerminalWindow::_RefreshThemeRoutine()
    {
        // Propagate the event to the host layer, so it can update its own UI
        _RequestedThemeChangedHandlers(*this, Theme());
    }

    winrt::fire_and_forget TerminalWindow::UpdateSettings(winrt::TerminalApp::SettingsLoadEventArgs args)
    {
        _settings = args.NewSettings();
        // Update the settings in TerminalPage
        _root->SetSettings(_settings, true);

        co_await wil::resume_foreground(_root->Dispatcher());

        // Bubble the notification up to the AppHost, now that we've updated our _settings.
        _SettingsChangedHandlers(*this, args);

        if (FAILED(args.Result()))
        {
            const winrt::hstring titleKey = USES_RESOURCE(L"ReloadJsonParseErrorTitle");
            const winrt::hstring textKey = USES_RESOURCE(L"ReloadJsonParseErrorText");
            _ShowLoadErrorsDialog(titleKey,
                                  textKey,
                                  gsl::narrow_cast<HRESULT>(args.Result()),
                                  args.ExceptionText());
            co_return;
        }
        else if (args.Result() == S_FALSE)
        {
            _ShowLoadWarningsDialog(args.Warnings());
        }
        _RefreshThemeRoutine();
    }

    void TerminalWindow::_OpenSettingsUI()
    {
        _root->OpenSettingsUI();
    }
    UIElement TerminalWindow::GetRoot() noexcept
    {
        return _root.as<winrt::Windows::UI::Xaml::Controls::Control>();
    }

    // Method Description:
    // - Gets the title of the currently focused terminal control. If there
    //   isn't a control selected for any reason, returns "Terminal"
    // Arguments:
    // - <none>
    // Return Value:
    // - the title of the focused control if there is one, else "Terminal"
    hstring TerminalWindow::Title()
    {
        if (_root)
        {
            return _root->Title();
        }
        return { L"Terminal" };
    }

    // Method Description:
    // - Used to tell the app that the titlebar has been clicked. The App won't
    //   actually receive any clicks in the titlebar area, so this is a helper
    //   to clue the app in that a click has happened. The App will use this as
    //   a indicator that it needs to dismiss any open flyouts.
    // Arguments:
    // - <none>
    // Return Value:
    // - <none>
    void TerminalWindow::TitlebarClicked()
    {
        if (_root)
        {
            _root->TitlebarClicked();
        }
    }

    // Method Description:
    // - Used to tell the PTY connection that the window visibility has changed.
    //   The underlying PTY might need to expose window visibility status to the
    //   client application for the `::GetConsoleWindow()` API.
    // Arguments:
    // - showOrHide - True is show; false is hide.
    // Return Value:
    // - <none>
    void TerminalWindow::WindowVisibilityChanged(const bool showOrHide)
    {
        if (_root)
        {
            _root->WindowVisibilityChanged(showOrHide);
        }
    }

    // Method Description:
    // - Implements the F7 handler (per GH#638)
    // - Implements the Alt handler (per GH#6421)
    // Return value:
    // - whether the key was handled
    bool TerminalWindow::OnDirectKeyEvent(const uint32_t vkey, const uint8_t scanCode, const bool down)
    {
        if (_root)
        {
            // Manually bubble the OnDirectKeyEvent event up through the focus tree.
            auto xamlRoot{ _root->XamlRoot() };
            auto focusedObject{ Windows::UI::Xaml::Input::FocusManager::GetFocusedElement(xamlRoot) };
            do
            {
                if (auto keyListener{ focusedObject.try_as<IDirectKeyListener>() })
                {
                    if (keyListener.OnDirectKeyEvent(vkey, scanCode, down))
                    {
                        return true;
                    }
                    // otherwise, keep walking. bubble the event manually.
                }

                if (auto focusedElement{ focusedObject.try_as<Windows::UI::Xaml::FrameworkElement>() })
                {
                    focusedObject = focusedElement.Parent();

                    // Parent() seems to return null when the focusedElement is created from an ItemTemplate.
                    // Use the VisualTreeHelper's GetParent as a fallback.
                    if (!focusedObject)
                    {
                        focusedObject = winrt::Windows::UI::Xaml::Media::VisualTreeHelper::GetParent(focusedElement);
                    }
                }
                else
                {
                    break; // we hit a non-FE object, stop bubbling.
                }
            } while (focusedObject);
        }
        return false;
    }

    // Method Description:
    // - Used to tell the app that the 'X' button has been clicked and
    //   the user wants to close the app. We kick off the close warning
    //   experience.
    // Arguments:
    // - <none>
    // Return Value:
    // - <none>
    void TerminalWindow::CloseWindow(LaunchPosition pos)
    {
        if (_root)
        {
            // If persisted layout is enabled and we are the last window closing
            // we should save our state.
            if (_settings.GlobalSettings().ShouldUsePersistedLayout() && _numOpenWindows == 1)
            {
                if (const auto layout = _root->GetWindowLayout())
                {
                    layout.InitialPosition(pos);
                    const auto state = ApplicationState::SharedInstance();
                    state.PersistedWindowLayouts(winrt::single_threaded_vector<WindowLayout>({ layout }));
                }
            }

            _root->CloseWindow(false);
        }
    }

    winrt::TerminalApp::TaskbarState TerminalWindow::TaskbarState()
    {
        if (_root)
        {
            return _root->TaskbarState();
        }
        return {};
    }

    winrt::Windows::UI::Xaml::Media::Brush TerminalWindow::TitlebarBrush()
    {
        if (_root)
        {
            return _root->TitlebarBrush();
        }
        return { nullptr };
    }
    void TerminalWindow::WindowActivated(const bool activated)
    {
        if (_root)
        {
            _root->WindowActivated(activated);
        }
    }

    // Method Description:
    // - Returns true if we should exit the application before even starting the
    //   window. We might want to do this if we're displaying an error message or
    //   the version string, or if we want to open the settings file.
    // Arguments:
    // - <none>
    // Return Value:
    // - true iff we should exit the application before even starting the window
    bool TerminalWindow::ShouldExitEarly()
    {
        return _appArgs.ShouldExitEarly();
    }

    bool TerminalWindow::FocusMode() const
    {
        return _root ? _root->FocusMode() : false;
    }

    bool TerminalWindow::Fullscreen() const
    {
        return _root ? _root->Fullscreen() : false;
    }

    void TerminalWindow::Maximized(bool newMaximized)
    {
        if (_root)
        {
            _root->Maximized(newMaximized);
        }
    }

    bool TerminalWindow::AlwaysOnTop() const
    {
        return _root ? _root->AlwaysOnTop() : false;
    }

    ////////////////////////////////////////////////////////////////////////////
    void TerminalWindow::SetSettingsStartupArgs(const std::vector<ActionAndArgs>& actions)
    {
        for (const auto& action : actions)
        {
            _settingsStartupArgs.push_back(action);
        }
        _gotSettingsStartupActions = true;
    }

    bool TerminalWindow::HasCommandlineArguments() const noexcept
    {
        return _hasCommandLineArguments;
    }

    // Method Description:
    // - Sets the initial commandline to process on startup, and attempts to
    //   parse it. Commands will be parsed into a list of ShortcutActions that
    //   will be processed on TerminalPage::Create().
    // - This function will have no effective result after Create() is called.
    // - This function returns 0, unless a there was a non-zero result from
    //   trying to parse one of the commands provided. In that case, no commands
    //   after the failing command will be parsed, and the non-zero code
    //   returned.
    // Arguments:
    // - args: an array of strings to process as a commandline. These args can contain spaces
    // Return Value:
    // - the result of the first command who's parsing returned a non-zero code,
    //   or 0. (see TerminalWindow::_ParseArgs)
    int32_t TerminalWindow::SetStartupCommandline(array_view<const winrt::hstring> args)
    {
        // This is called in AppHost::ctor(), before we've created the window
        // (or called TerminalWindow::Initialize)
        const auto result = _appArgs.ParseArgs(args);
        if (result == 0)
        {
            // If the size of the arguments list is 1,
            // then it contains only the executable name and no other arguments.
            _hasCommandLineArguments = args.size() > 1;
            _appArgs.ValidateStartupCommands();

            // DON'T pass the args into the page yet. It doesn't exist yet.
            // Instead, we'll handle that in Initialize, when we first instantiate the page.
        }

        // If we have a -s param passed to us to load a saved layout, cache that now.
        if (const auto idx = _appArgs.GetPersistedLayoutIdx())
        {
            SetPersistedLayoutIdx(idx.value());
        }
        return result;
    }

    // Method Description:
    // - Parse the provided commandline arguments into actions, and try to
    //   perform them immediately.
    // - This function returns 0, unless a there was a non-zero result from
    //   trying to parse one of the commands provided. In that case, no commands
    //   after the failing command will be parsed, and the non-zero code
    //   returned.
    // - If a non-empty cwd is provided, the entire terminal exe will switch to
    //   that CWD while we handle these actions, then return to the original
    //   CWD.
    // Arguments:
    // - args: an array of strings to process as a commandline. These args can contain spaces
    // - cwd: The directory to use as the CWD while performing these actions.
    // Return Value:
    // - the result of the first command who's parsing returned a non-zero code,
    //   or 0. (see TerminalWindow::_ParseArgs)
    int32_t TerminalWindow::ExecuteCommandline(array_view<const winrt::hstring> args,
                                               const winrt::hstring& cwd)
    {
        ::TerminalApp::AppCommandlineArgs appArgs;
        auto result = appArgs.ParseArgs(args);
        if (result == 0)
        {
            auto actions = winrt::single_threaded_vector<ActionAndArgs>(std::move(appArgs.GetStartupActions()));

            _root->ProcessStartupActions(actions, false, cwd);

            if (appArgs.IsHandoffListener())
            {
                _root->SetInboundListener(true);
            }
        }
        // Return the result of parsing with commandline, though it may or may not be used.
        return result;
    }

    // Method Description:
    // - If there were any errors parsing the commandline that was used to
    //   initialize the terminal, this will return a string containing that
    //   message. If there were no errors, this message will be blank.
    // - If the user requested help on any command (using --help), this will
    //   contain the help message.
    // - If the user requested the version number (using --version), this will
    //   contain the version string.
    // Arguments:
    // - <none>
    // Return Value:
    // - the help text or error message for the provided commandline, if one
    //   exists, otherwise the empty string.
    winrt::hstring TerminalWindow::ParseCommandlineMessage()
    {
        return winrt::to_hstring(_appArgs.GetExitMessage());
    }

    ////////////////////////////////////////////////////////////////////////////

    hstring TerminalWindow::GetWindowLayoutJson(LaunchPosition position)
    {
        if (_root != nullptr)
        {
            if (const auto layout = _root->GetWindowLayout())
            {
                layout.InitialPosition(position);
                return WindowLayout::ToJson(layout);
            }
        }
        return L"";
    }

    void TerminalWindow::SetPersistedLayoutIdx(const uint32_t idx)
    {
        _loadFromPersistedLayoutIdx = idx;
    }

    // Method Description;
    // - Checks if the current window is configured to load a particular layout
    // Arguments:
    // - settings: The settings to use as this may be called before the page is
    //   fully initialized.
    // Return Value:
    // - non-null if there is a particular saved layout to use
    std::optional<uint32_t> TerminalWindow::LoadPersistedLayoutIdx() const
    {
        return _settings.GlobalSettings().ShouldUsePersistedLayout() ? _loadFromPersistedLayoutIdx : std::nullopt;
    }

    WindowLayout TerminalWindow::LoadPersistedLayout() const
    {
        if (const auto idx = LoadPersistedLayoutIdx())
        {
            const auto i = idx.value();
            const auto layouts = ApplicationState::SharedInstance().PersistedWindowLayouts();
            if (layouts && layouts.Size() > i)
            {
                return layouts.GetAt(i);
            }
        }
        return nullptr;
    }

    void TerminalWindow::SetNumberOfOpenWindows(const uint64_t num)
    {
        _numOpenWindows = num;
        if (_root)
        {
            _root->SetNumberOfOpenWindows(num);
        }
    }
    ////////////////////////////////////////////////////////////////////////////
    void TerminalWindow::RequestExitFullscreen()
    {
        _root->SetFullscreen(false);
    }

    bool TerminalWindow::AutoHideWindow()
    {
        return _settings.GlobalSettings().AutoHideWindow();
    }

    void TerminalWindow::UpdateSettingsHandler(const winrt::IInspectable& /*sender*/,
                                               const winrt::TerminalApp::SettingsLoadEventArgs& args)
    {
        UpdateSettings(args);
    }

    ////////////////////////////////////////////////////////////////////////////

    void TerminalWindow::IdentifyWindow()
    {
        if (_root)
        {
            _root->IdentifyWindow();
        }
    }

    void TerminalWindow::RenameFailed()
    {
        if (_root)
        {
            _root->RenameFailed();
        }
    }

    winrt::hstring TerminalWindow::WindowName() const noexcept
    {
        return _WindowName;
    }
    void TerminalWindow::WindowName(const winrt::hstring& name)
    {
        const auto oldIsQuakeMode = IsQuakeWindow();

        const auto changed = _WindowName != name;
        if (changed)
        {
            _WindowName = name;
            if (_root)
            {
                _root->WindowNameChanged();

                // If we're entering quake mode, or leaving it
                if (IsQuakeWindow() != oldIsQuakeMode)
                {
                    // If we're entering Quake Mode from ~Focus Mode, then this will enter Focus Mode
                    // If we're entering Quake Mode from Focus Mode, then this will do nothing
                    // If we're leaving Quake Mode (we're already in Focus Mode), then this will do nothing
                    _root->SetFocusMode(true);
                    _IsQuakeWindowChangedHandlers(*this, nullptr);
                }
            }
        }
    }
    uint64_t TerminalWindow::WindowId() const noexcept
    {
        return _WindowId;
    }
    void TerminalWindow::WindowId(const uint64_t& id)
    {
        if (_WindowId != id)
        {
            _WindowId = id;
            if (_root)
            {
                _root->WindowNameChanged();
            }
        }
    }

    // Method Description:
    // - Returns a label like "Window: 1234" for the ID of this window
    // Arguments:
    // - <none>
    // Return Value:
    // - a string for displaying the name of the window.
    winrt::hstring TerminalWindow::WindowIdForDisplay() const noexcept
    {
        return winrt::hstring{ fmt::format(L"{}: {}",
                                           std::wstring_view(RS_(L"WindowIdLabel")),
                                           _WindowId) };
    }

    // Method Description:
    // - Returns a label like "<unnamed window>" when the window has no name, or the name of the window.
    // Arguments:
    // - <none>
    // Return Value:
    // - a string for displaying the name of the window.
    winrt::hstring TerminalWindow::WindowNameForDisplay() const noexcept
    {
        return _WindowName.empty() ?
                   winrt::hstring{ fmt::format(L"<{}>", RS_(L"UnnamedWindowName")) } :
                   _WindowName;
    }
    bool TerminalWindow::IsQuakeWindow() const noexcept
    {
        return _WindowName == QuakeWindowName;
    }
    ////////////////////////////////////////////////////////////////////////////

    bool TerminalWindow::ShouldImmediatelyHandoffToElevated()
    {
        return _root != nullptr ? _root->ShouldImmediatelyHandoffToElevated(_settings) : false;
    }

    // Method Description:
    // - Escape hatch for immediately dispatching requests to elevated windows
    //   when first launched. At this point in startup, the window doesn't exist
    //   yet, XAML hasn't been started, but we need to dispatch these actions.
    //   We can't just go through ProcessStartupActions, because that processes
    //   the actions async using the XAML dispatcher (which doesn't exist yet)
    // - DON'T CALL THIS if you haven't already checked
    //   ShouldImmediatelyHandoffToElevated. If you're thinking about calling
    //   this outside of the one place it's used, that's probably the wrong
    //   solution.
    // Arguments:
    // - settings: the settings we should use for dispatching these actions. At
    //   this point in startup, we hadn't otherwise been initialized with these,
    //   so use them now.
    // Return Value:
    // - <none>
    void TerminalWindow::HandoffToElevated()
    {
        if (_root)
        {
            _root->HandoffToElevated(_settings);
            return;
        }
    }

};
